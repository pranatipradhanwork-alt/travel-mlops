// Objective 6: CI/CD for the travel price prediction service.
// Required Jenkins credentials:
//   kubeconfig             (secret file)        - deploy to the cluster
//   docker-registry-creds  (username/password)  - push images (only when REGISTRY != 'local')
// REGISTRY=local builds images straight into the cluster's Docker engine (Colima/k3s,
// minikube docker-env) - no push needed. See jenkins/README.md for the local setup.
pipeline {
    agent any

    parameters {
        string(name: 'REGISTRY', defaultValue: 'local',
               description: "'local' = cluster shares Jenkins' Docker engine (no push); else e.g. docker.io/<user>")
        booleanParam(name: 'DEPLOY', defaultValue: true, description: 'Deploy to Kubernetes after a green build')
        booleanParam(name: 'RETRAIN', defaultValue: true, description: 'Retrain models (false = reuse models/ from workspace)')
    }

    environment {
        IMAGE_TAG = "${env.BUILD_NUMBER}-${env.GIT_COMMIT ? env.GIT_COMMIT.take(7) : 'local'}"
        IMAGE_PREFIX = "${params.REGISTRY == 'local' ? '' : params.REGISTRY + '/'}"
        API_IMAGE = "${IMAGE_PREFIX}travel-price-api"
        UI_IMAGE  = "${IMAGE_PREFIX}travel-streamlit"
        MLFLOW_DISABLE_AGENT_HINT = '1'
        VENV = "${WORKSPACE}/.venv"
        PIP_CACHE_DIR = "${JENKINS_HOME}/.pip-cache"
    }

    options {
        timestamps()
        timeout(time: 60, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20'))
        disableConcurrentBuilds()
    }

    stages {
        stage('Checkout') {
            steps { checkout scm }
        }

        stage('Setup') {
            steps {
                sh '''
                    python3 -m venv "$VENV"
                    "$VENV/bin/pip" install -q --upgrade pip
                    "$VENV/bin/pip" install -q -r requirements-dev.txt
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '"$VENV/bin/flake8" src api tests streamlit_app airflow/dags --max-line-length 120'
            }
        }

        stage('Data validation') {
            steps {
                // Fails the build on schema/quality problems; drift report is archived for review.
                sh '''
                    "$VENV/bin/python" -m src.validate_cli
                    "$VENV/bin/python" -m src.monitor
                '''
            }
        }

        stage('Train & track (MLflow)') {
            when { expression { params.RETRAIN } }
            steps {
                // Each trainer exits non-zero if its quality gate fails, which fails the build.
                sh '"$VENV/bin/python" -m src.pipeline'
            }
            post {
                always { archiveArtifacts artifacts: 'models/metrics/*.json, reports/**', allowEmptyArchive: true }
            }
        }

        stage('Unit & API tests') {
            steps {
                sh '"$VENV/bin/python" -m pytest -q --junitxml=reports/junit.xml'
            }
            post {
                always { junit testResults: 'reports/junit.xml', allowEmptyResults: true }
            }
        }

        stage('Build images') {
            parallel {
                stage('API') {
                    steps { sh 'docker build -t "$API_IMAGE:$IMAGE_TAG" -t "$API_IMAGE:latest" .' }
                }
                stage('Streamlit') {
                    steps { sh 'docker build -f Dockerfile.streamlit -t "$UI_IMAGE:$IMAGE_TAG" -t "$UI_IMAGE:latest" .' }
                }
            }
        }

        stage('Container smoke test') {
            steps {
                sh '''
                    docker rm -f api-smoke >/dev/null 2>&1 || true
                    docker run -d --name api-smoke -p 15000:8000 "$API_IMAGE:$IMAGE_TAG"
                    for i in $(seq 1 30); do
                        curl -fsS http://localhost:15000/health && break
                        sleep 2
                    done
                    curl -fsS -X POST http://localhost:15000/predict -H 'Content-Type: application/json' \
                      -d '{"from":"Recife (PE)","to":"Florianopolis (SC)","flightType":"economic","agency":"Rainbow"}'
                '''
            }
            post {
                always { sh 'docker rm -f api-smoke >/dev/null 2>&1 || true' }
            }
        }

        stage('Push images') {
            when { expression { params.DEPLOY && params.REGISTRY != 'local' } }
            steps {
                withCredentials([usernamePassword(credentialsId: 'docker-registry-creds',
                                                  usernameVariable: 'REG_USER', passwordVariable: 'REG_PASS')]) {
                    sh '''
                        echo "$REG_PASS" | docker login "${REGISTRY%%/*}" -u "$REG_USER" --password-stdin
                        docker push "$API_IMAGE:$IMAGE_TAG" && docker push "$API_IMAGE:latest"
                        docker push "$UI_IMAGE:$IMAGE_TAG"  && docker push "$UI_IMAGE:latest"
                    '''
                }
            }
        }

        stage('Deploy to Kubernetes') {
            when { expression { params.DEPLOY } }
            steps {
                withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
                    sh '''
                        kubectl apply -k k8s/
                        kubectl -n travel-ml set image deployment/travel-price-api api="$API_IMAGE:$IMAGE_TAG"
                        kubectl -n travel-ml set image deployment/travel-streamlit streamlit="$UI_IMAGE:$IMAGE_TAG"
                        if ! kubectl -n travel-ml rollout status deployment/travel-price-api --timeout=180s; then
                            echo "Rollout failed - rolling back"
                            kubectl -n travel-ml rollout undo deployment/travel-price-api
                            exit 1
                        fi
                        kubectl -n travel-ml rollout status deployment/travel-streamlit --timeout=180s
                    '''
                }
            }
        }
    }

    post {
        success { echo "Deployed ${API_IMAGE}:${IMAGE_TAG}" }
        failure { echo 'Pipeline failed - production left on previous version.' }
        always  { sh 'docker image prune -f >/dev/null 2>&1 || true' }
    }
}
