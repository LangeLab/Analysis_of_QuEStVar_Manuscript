# Small bash script to convert all notebooks within 
#   a folder and its subfolders to an html.
# This html convert uses the template from pretty-jupyter
# pretty-jupyter: https://github.com/JanPalasek/pretty-jupyter

# Usage: bash nb_to_html.sh

# Start timer
start=$(date +%s.%N)

for i in $(find . -name "*.ipynb"); do
    jupyter nbconvert --to html --template pj $i
done

# Collect all html files to a folder
mkdir supp_notebooks
# Initialize .init file
touch supp_notebooks/.init
# Move all html files to the folder
mv **/*.html supp_notebooks

# End timer
end=$(date +%s.%N)

# Print time
echo "Time elapsed: " $(echo "$end - $start" | bc) "seconds"
