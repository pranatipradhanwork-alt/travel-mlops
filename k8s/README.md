# Kubernetes Deployment

| File | What it creates |
|---|---|
| `namespace.yaml` | `travel-ml` namespace |
| `api-deployment.yaml` | API Deployment (2 replicas, rolling update with `maxUnavailable: 0`, readiness/liveness probes on `/health`, CPU/memory requests and limits, non-root user) + ConfigMap + ClusterIP Service |
| `api-hpa.yaml` | HorizontalPodAutoscaler (2–10 pods, CPU 60% / memory 80%) + PodDisruptionBudget (`minAvailable: 1`) |
| `streamlit.yaml` | Streamlit Deployment + Service. Calls the API through cluster DNS. |
| `ingress.yaml` | `api.travel.local` → API, `travel.local` → Streamlit |
| `kustomization.yaml` | Ties everything together. Jenkins sets the image tags. |

## Local walkthrough — Colima (recommended, used for the verified run)

Colima runs Docker and a k3s Kubernetes in one small VM. k3s uses the same Docker engine, so locally built
images are visible to the cluster without a registry. metrics-server (needed by the HPA) is included.

```bash
brew install colima docker docker-buildx kubectl
make colima-up                            # 2 CPU / 4 GB VM with Kubernetes
make k8s-deploy                           # build images + kubectl apply -k k8s/ + wait for rollout
kubectl -n travel-ml get pods,svc,hpa,pdb
make k8s-forward                          # API → localhost:8000, Streamlit → localhost:8501
curl localhost:8000/health
```

### Alternative — minikube
```bash
minikube start --cpus 4 --memory 6g
minikube addons enable metrics-server
make minikube-deploy                      # builds images, loads them into minikube, applies k8s/
```
The Ingress (`k8s/ingress.yaml`) is optional. Apply it after enabling an nginx ingress controller
(`minikube addons enable ingress`).

## Watch it scale
```bash
kubectl -n travel-ml get hpa -w                         # terminal 1
kubectl -n travel-ml apply -f k8s/loadgen.yaml          # terminal 2: 16 threads hitting the Service for 150 s
kubectl -n travel-ml delete -f k8s/loadgen.yaml         # clean up afterwards
```
The load generator runs **inside** the cluster (built from the API image, so there's nothing extra to download) and goes
through the Service, so load is spread across pods.

**Verified run (Colima, 2 CPU node):**

| Time | CPU (target 60%) | Replicas |
|---|---|---|
| 0 s | 3% | 2 |
| 20 s | 40% | 10 |
| 40 s | 344% | 10 |
| 160 s | 113% | 10 |
| 180 s (load stopped) | 1% | 10 → back to 2 after the 5-min stabilisation window |

* The HPA scaled from 2 to 10 pods within 40 s, and the cluster served 35,087 batch requests (60 predictions each) in 150 s.
* Six pods ran and four stayed `Pending` with *Insufficient cpu*, because the single 2-CPU node's requests were 95%
  allocated. In a cloud cluster, the **Cluster Autoscaler** would add nodes at this point, which is the next layer of scaling.

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
