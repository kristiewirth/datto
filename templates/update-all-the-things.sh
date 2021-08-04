'''
Modified from this original source: 
https://medium.com/@waxzce/keeping-macos-clean-this-is-my-osx-brew-update-cli-command-6c8f12dc1731
'''

# Install new versions of outdated brew packages
brew upgrade

# Keep only latest versions
brew cleanup -s

# Upgrade all Atom plugins
apm upgrade -c false

# Upgrade all VS Code plugins
code --list-extensions | xargs -n1 code --install-extension -U

# Upgrade Mac apps from app store (need to `brew install mas` first)
mas upgrade

# Update all pip
pip install --upgrade pip | pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U

# Remove files & folders older than 30 days in Downloads folder
# WARNING - This will delete files automatically, remove if you don't want to do this!
find Downloads/ -ctime +30 -print0 | xargs -0 rm -r

# Clean up Docker
docker system prune -a -f

# Check if anything was broken with pip
echo "---------------------------------------------------------------------------"
echo "PIP CHECK RESULTS:"
pip check
echo "---------------------------------------------------------------------------"
