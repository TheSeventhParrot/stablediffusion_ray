### Pre-requisites
1) I'm installing on a Debian 11 machine deployed on GCE (Utilizing the ```c0-deeplearning-common-cpu-v20241224-debian-11``` image)
2) Make sure to install the appropriate cuda package with
   ```wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb```
   ```sudo dpkg -i cuda-keyring_1.0-1_all.deb```
   ```sudo apt-get update```
   ```sudo apt-get install nvidia-driver```
   ```sudo reboot```
3) Install nvidia container toolkit with
   ```distribution=$(. /etc/os-release; echo $ID$VERSION_ID)```
   ```echo $distribution```
   ```curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg```
   ```curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list```
   ```sudo apt-get update```
   ```sudo apt-get install -y nvidia-container-toolkit```
4) Set up a GKE cluster with a gpu pool of 2 nodes and a singular head node
5) You need to also install helm and appropriate resources which can be done like so:
   ```curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null```
   ```echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list```
   ```sudo apt-get update```
   ```sudo apt-get install helm```
   ```helm repo add kuberay https://ray-project.github.io/kuberay-helm/```
   ```helm repo update```
   ```helm install kuberay-operator kuberay/kuberay-operator```
   ```helm install raycluster kuberay/ray-cluster```
6) Your GKE cluster is ready if it looks like this after running ```kubectl get pods```
   kuberay-operator-5dd6779f94-hrcwl             1/1     Running   0             4h43m
   raycluster-kuberay-head-nqljc                 1/1     Running   0             44m
   raycluster-kuberay-workergroup-worker-j8nq8   1/1     Running   1             103m
   raycluster-kuberay-workergroup-worker-rl6jz   1/1     Running   1             103m
   

### Usecase
1) This examples works on finetuning stable diffusion 2 to create Lovecraftian monsters
2) This can be altered to create other styles of images as well as utilizing different stable diffusion models
3) I'm utilizing GKE to create a cluster
