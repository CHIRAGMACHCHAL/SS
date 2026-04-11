#!/bin/bash
# SS Mini AGI - Setup Script
# Automated setup for the Vedic Intelligence System

set -e  # Exit on error

echo "=========================================="
echo "SS Mini AGI - Setup Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}Error: Python 3.8+ is required. Found: $python_version${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version detected${NC}"

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping.${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Download spaCy model
echo -e "\n${YELLOW}Downloading spaCy model...${NC}"
python -m spacy download en_core_web_sm
echo -e "${GREEN}✓ spaCy model downloaded${NC}"

# Create .env file if it doesn't exist
echo -e "\n${YELLOW}Setting up environment variables...${NC}"
if [ -f ".env" ]; then
    echo -e "${YELLOW}.env file already exists. Skipping.${NC}"
else
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created from template${NC}"
    echo -e "${YELLOW}⚠ Please edit .env file with your actual configuration${NC}"
fi

# Create data directories
echo -e "\n${YELLOW}Creating data directories...${NC}"
mkdir -p data/scriptures/public
mkdir -p data/scriptures/private
mkdir -p data/training/datasets
mkdir -p logs
echo -e "${GREEN}✓ Data directories created${NC}"

# Check for required services
echo -e "\n${YELLOW}Checking required services...${NC}"

# Check PostgreSQL
if command -v psql &> /dev/null; then
    echo -e "${GREEN}✓ PostgreSQL found${NC}"
else
    echo -e "${YELLOW}⚠ PostgreSQL not found. Please install it separately.${NC}"
fi

# Check Redis
if command -v redis-cli &> /dev/null; then
    echo -e "${GREEN}✓ Redis found${NC}"
else
    echo -e "${YELLOW}⚠ Redis not found. Please install it separately.${NC}"
fi

# Check Docker (for Qdrant)
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker found${NC}"
    
    # Offer to start Qdrant
    read -p "Would you like to start Qdrant with Docker? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Starting Qdrant...${NC}"
        docker run -d -p 6333:6333 -p 6334:6334 \
            -v $(pwd)/data/qdrant_storage:/qdrant/storage \
            qdrant/qdrant
        echo -e "${GREEN}✓ Qdrant started${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Docker not found. Please install it to run Qdrant.${NC}"
fi

# Setup complete
echo -e "\n${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "1. Edit .env file with your configuration"
echo "2. Add scripture PDFs to data/scriptures/"
echo "3. Ensure PostgreSQL, Redis, and Qdrant are running"
echo "4. Run the demo: python demo.py interactive"

echo -e "\n${YELLOW}Useful Commands:${NC}"
echo "  Activate venv:  source venv/bin/activate"
echo "  Run demo:       python demo.py interactive"
echo "  Start API:      python api/custom_api.py"
echo "  Run tests:      python -m pytest tests/"

echo -e "\n${GREEN}Happy coding! 🙏${NC}"
