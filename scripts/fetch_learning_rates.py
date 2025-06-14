def extract_learning_rates(log_file_path):
    learning_rates = []
    
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                # Look for learning_rate pattern in the line
                if 'learning_rate:' in line:
                    # Extract the learning rate value after 'learning_rate:'
                    lr_part = line.split('learning_rate:')[1].strip()
                    # Handle scientific notation and regular floats
                    try:
                        lr_value = float(lr_part)
                        learning_rates.append(lr_value)
                    except ValueError:
                        continue
                
                # Handle case where it's just a single learning rate value (like B0)
                elif line.replace('.', '').replace('e', '').replace('-', '').replace('+', '').isdigit():
                    try:
                        lr_value = float(line)
                        learning_rates.append(lr_value)
                    except ValueError:
                        continue
        
        return learning_rates
        
    except FileNotFoundError:
        print(f"File not found: {log_file_path}")
        return []
    except Exception as e:
        print(f"Error reading file {log_file_path}: {str(e)}")
        return []

# Test the function
if __name__ == "__main__":
    # Test with your files
    test_files = [
        "/home/veysel/dev-projects/StayAwake-AI/models_inattention/B0_without_normalized_images_16_batches/learning_rates.txt",
    ]
    
    for file_path in test_files:
        rates = extract_learning_rates(file_path)
        print(f"{file_path}: {rates}")