param(
    [Parameter(Position=0)]
    [ValidateSet("up", "local", "down", "logs")]
    [string]$Command = "up"
)

switch ($Command) {
    "up" {
        Write-Host "Avvio stack con Ollama esterno..." -ForegroundColor Cyan
        docker compose up -d --build
    }
    "local" {
        Write-Host "Avvio stack con Ollama locale..." -ForegroundColor Cyan
        $env:OLLAMA_URL = "http://ollama:11434/api/generate"
        docker compose --profile local up -d --build
        Remove-Item Env:OLLAMA_URL -ErrorAction SilentlyContinue
    }
    "down" {
        Write-Host "Arresto di tutti i container..." -ForegroundColor Yellow
        docker compose --profile local down
    }
    "logs" {
        docker compose --profile local logs -f
    }
}
