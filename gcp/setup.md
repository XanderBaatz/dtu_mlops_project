## Setup tutorial

### Login
Login/authenticate:
```bash
gcloud auth login
```

### Set project
List projects:
```bash
gcloud projects list
```

Set project property:
```bash
gcloud config set project dtumlops-484312
```

### Enable services
Enable services:
```bash
gcloud services enable apigateway.googleapis.com
gcloud services enable servicemanagement.googleapis.com
gcloud services enable servicecontrol.googleapis.com
```

### Service Account
Permissions:
- Storage Object Viewer
- Cloud Build Builder
- Secret Manager Secret Accessor
- AI Platform Developer
- Artifact Registry Writer

### Connect to VM instance
List VM instances:
```bash
gcloud compute instances list
```

Connect to instance via terminal
```bash
gcloud compute ssh --zone europe-west1-d pytorch-instance --project dtumlops-484312
```

### List pytorch enabled instances
Command:
```bash
gcloud container images list --repository="gcr.io/deeplearning-platform-release" --filter="name~'pytorch'"
```

Check GPU (?) enabled images:
```bash
gcloud compute images list --project deeplearning-platform-release --no-standard-images --filter="name~'pytorch'"
```

Should list:
```
NAME                                                PROJECT                        FAMILY                                    DEPRECATED  STATUS
pytorch-2-7-cu128-ubuntu-2204-nvidia-570-v20260120  deeplearning-platform-release  pytorch-2-7-cu128-ubuntu-2204-nvidia-570              READY
pytorch-2-7-cu128-ubuntu-2404-nvidia-570-v20260120  deeplearning-platform-release  pytorch-2-7-cu128-ubuntu-2404-nvidia-570              READY
```

Possible working command:
```bash
gcloud compute instances create pytorch-instance-gpu \
    --zone="europe-west1-b" \
    --image-family="pytorch-2-7-cu128-ubuntu-2404-nvidia-570" \
    --image-project="deeplearning-platform-release" \
    #--accelerator="type=nvidia-tesla-t4,count=1" \
    #--maintenance-policy TERMINATE
```

### Artifact Registry
```bash
gcloud artifacts repositories create docker-artifact \
    --repository-format=docker \
    --location=europe-west1 \
    --description="My docker registry"
```

Policy:
```bash
gcloud artifacts repositories set-cleanup-policies docker-artifact \
    --project=dtumlops-484312 \
    --location=europe-west1 \
    --policy=gcp/policy.yaml
```

Pull and push to this artifact repository (URL):
```bash
europe-west1-docker.pkg.dev/dtumlops-484312/docker-artifact
```

### Build image
```bash
gcloud builds submit . --config=gcp/cloudbuild.yaml --substitutions=_IMAGE_NAME=train
```
