# table texture test
# rm /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png
# cp /home/users/ghc/zzy/act-plus-plus/assets/table_texture_gen0.png /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png

# rm /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png
# cp /home/users/ghc/zzy/act-plus-plus/assets/table_texture_gen1.png /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png

# rm /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png
# cp /home/users/ghc/zzy/act-plus-plus/assets/table_texture_gen2.png /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png

# rm /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png
# cp /home/users/ghc/zzy/act-plus-plus/assets/table_texture_gen3.png /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png



process_files() {
    local target_dir=$1
    local gen_number=$2
    for file in "$target_dir"/*_gen${gen_number}.xml; 
    do
        base_name=$(basename "$file" "_gen${gen_number}.xml")
        non_gen_file="$target_dir/$base_name.xml"
        if [ -f "$non_gen_file" ]; then
            echo "Deleting $non_gen_file"
            rm "$non_gen_file"
        fi
        new_xml_file="$target_dir/$base_name.xml"
        echo "Copying $file to $new_xml_file"
        cp "$file" "$new_xml_file"
    done
}

TARGET_DIR="/home/users/ghc/zzy/act-plus-plus/assets"
GEN_NUMBER=0

process_files "$TARGET_DIR" "$GEN_NUMBER"




