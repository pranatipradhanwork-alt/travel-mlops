# Local Jenkins

Runs the repository's [`Jenkinsfile`](../Jenkinsfile) end to end on your machine: lint → data validation →
training with MLflow and quality gates → 24 tests → image builds → container smoke test → deploy to
Kubernetes.

| File | Purpose |
|---|---|
| `Dockerfile` | Jenkins LTS (JDK 21) + Python 3, Docker CLI (+ buildx) and kubectl |
| `plugins.txt` | Pipeline, Git, Job DSL, Configuration as Code, credentials, JUnit, stage view |
| `casc.yaml` | Configuration as Code: skips the setup wizard, creates the `kubeconfig` credential and the `travel-mlops` pipeline job (reads `Jenkinsfile` from the repo) |

> ⚠️ **Local demo only.** Security is disabled (`securityRealm: none`) so no login is needed. Never expose this
> instance to a network. For a shared server, configure users and authorization in `casc.yaml`.

## Start (Colima)

```bash
make colima-up        # Docker + Kubernetes VM (once per boot)
make jenkins-up       # builds the image, starts Jenkins on http://localhost:8080
make jenkins-build    # triggers the pipeline (REGISTRY=local, DEPLOY=true, RETRAIN=true)
```
Then open **http://localhost:8080/job/travel-mlops/** and click the running build → *Pipeline Overview* (or
*Stage View*) to watch each stage.

How it is wired:
* **Host network** inside the Colima VM, so Jenkins reaches the k3s API server and the smoke-test container on `localhost`.
* **The VM's Docker socket is mounted**, so images Jenkins builds are immediately available to k3s.
  `REGISTRY=local` skips the registry push.
* **The repository is mounted read-only at `/repo`**, and Jenkins checks out the committed `main` branch from it.
  **Commit your changes before triggering a build.**
* **The kubeconfig** is copied from the VM to `~/.travel-jenkins/kubeconfig` (mode 600, outside the repo) and
  mounted as the `kubeconfig` credential.

## Verified run (build #1)

| Stage | Result |
|---|---|
| Setup → Lint → Data validation | ✅ |
| Train & track (MLflow) | ✅ RF champion, gender acc 0.894, recommender hit@3 0.375 |
| Unit & API tests | ✅ 24 passed (JUnit report in Jenkins) |
| Build images (parallel) → Container smoke test | ✅ |
| Push images | skipped (`REGISTRY=local`) |
| Deploy to Kubernetes | ✅ `travel-price-api:1-<git sha>` rolled out |

Total time: 3 min 40 s.

## Using a real registry
Add a `docker-registry-creds` username/password credential and run the job with `REGISTRY=docker.io/<user>`.
The *Push images* stage then runs and the cluster pulls from the registry.

## Stop / clean up
```bash
docker stop jenkins            # keeps job history in the jenkins_home volume
docker rm -f jenkins && docker volume rm jenkins_home   # full reset
```
