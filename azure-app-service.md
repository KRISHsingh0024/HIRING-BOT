# Azure App Service (Linux) Deployment

This app is a single FastAPI service that also serves a static UI from `app/static/`.

## Option A (recommended): App Service using a Docker container

### 1) Create Azure resources

- Create a Resource Group
- Create an Azure Container Registry (ACR)
- Create an App Service Plan (Linux)
- Create a Web App for Containers

You can do this in the Azure Portal, or via Azure CLI.

### 2) Build and push the image to ACR

From the repo root:

```bash
az login
az group create -n <rg> -l <region>
az acr create -n <acrName> -g <rg> --sku Basic

az acr login -n <acrName>

# Build + push
az acr build -r <acrName> -t shl-nim-rag:latest .
```

### 3) Configure Web App to use the container

In the Web App:
- Container settings: `acrname.azurecr.io/shl-nim-rag:latest`
- Set environment variables (Configuration → Application settings):
  - `NVIDIA_API_KEY` (secret)
  - `NVIDIA_NIM_BASE_URL` = `https://integrate.api.nvidia.com/v1`
  - `AUTO_BUILD_INDICES` = `true`
  - `FAISS_INDEX_PATH` = `data/faiss.index`
  - `EMBEDDINGS_PKL` = `data/embeddings.pkl`
  - `BM25_PKL` = `data/bm25_retriever.pkl`
  - `CATALOG_JSON` = `data/catalog.json`

### 4) Ensure the app binds to the correct port

The container uses `PORT` automatically:
- App Service injects `PORT` and routes traffic to it.

### 5) Health check

Set App Service Health Check path to:
- `/health`

---

## Notes

- The embedding/index build step (`scripts/build_embeddings.py`) currently runs locally. For fully automated deployments, you have two choices:
  1) Commit generated `data/*.pkl` and `data/*.index` (not recommended; large + couples deploy to data)
  2) Add a startup step in the container entrypoint to build indices on first boot (recommended only if catalog is small and boot time is acceptable)

If you want, I can implement option (2) as a safe "build-if-missing" startup script.
