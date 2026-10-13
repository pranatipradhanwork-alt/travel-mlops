# Kubernetes Deployment

| File | What it creates |
|---|---|
| `namespace.yaml` | `travel-ml` namespace |
| `api-deployment.yaml` | API Deployment (2 replicas, rolling update with `maxUnavailable: 0`, readiness/liveness probes on `/health`, CPU/memory requests and limits, non-root user) + ConfigMap + ClusterIP Service |
| `api-hpa.yaml` | HorizontalPodAutoscaler (2–10 pods, CPU 60% / memory 80%) + PodDisruptionBudget (`minAvailable: 1`) |
| `streamlit.yaml` | Streamlit Deployment + Service. Calls the API through cluster DNS. |
| `ingress.yaml` | `api.travel.local` → API, `travel.local` → Streamlit |
| `kustomization.yaml` | Ties everything together. Jenkins sets the image tags. |

## Local walkthrough (minikube)

```bash
minikube start --cpus 4 --memory 6g
minikube addons enable metrics-server     # required by the HPA
minikube addons enable ingress
make minikube-deploy                      # builds images, loads them into minikube, applies k8s/
kubectl -n travel-ml get pods,svc,hpa
kubectl -n travel-ml port-forward svc/travel-price-api 8000:80
curl localhost:8000/health
```

## Watch it scale
```bash
kubectl -n travel-ml get hpa -w                         # terminal 1
# terminal 2: generate load (install `hey`: brew install hey)
hey -z 2m -c 50 -m POST -H 'Content-Type: application/json' \
  -d '{"from":"Recife (PE)","to":"Natal (RN)","flightType":"economic","agency":"Rainbow"}' \
  http://localhost:8000/predict
```
When average CPU goes above 60%, replicas scale up (they can double every 30 s). Scale-down waits 5 minutes
to avoid flapping.

## Why these settings
* **Readiness probe:** a pod only receives traffic after its models are loaded.
* **Liveness probe:** a hung worker gets restarted.
* **`maxUnavailable: 0` + PDB:** zero-downtime deploys and node drains.
* **Resource requests:** the HPA needs them to compute utilisation, and the scheduler uses them to place pods.
* **Image tag per build** (`<build>-<git sha>`): every running pod maps to one exact model version.

## Rollback
```bash
kubectl -n travel-ml rollout history deployment/travel-price-api
kubectl -n travel-ml rollout undo deployment/travel-price-api
```
Jenkins does this automatically if a rollout does not become healthy within 180 s.
