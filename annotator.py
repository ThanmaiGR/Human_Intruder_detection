import os


def update_labels(input_dir, output_dir, label_offset=15):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):  # Process only annotation files
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)

            with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
                for line in infile:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_label = int(parts[0])
                        updated_label = class_label - label_offset
                        outfile.write(f"{updated_label} {' '.join(parts[1:])}\n")


# Specify your input and output directories
input_dir = r'img for human intrusion/img for human intrusion/labels'
output_dir = r'img for human intrusion/new_labels'

# Update the labels
update_labels(input_dir, output_dir)
