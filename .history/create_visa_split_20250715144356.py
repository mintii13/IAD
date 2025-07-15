import os
import csv

def create_visa_split_csv(data_folder, output_csv):
    """
    Tạo file CSV split cho VisA dataset
    """
    data_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 
                 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    
    rows = []
    
    for data_class in data_list:
        class_folder = os.path.join(data_folder, data_class)
        
        if not os.path.exists(class_folder):
            print(f"Warning: {class_folder} not found, skipping...")
            continue
        
        # Duyệt qua train/test folders
        for set_type in ['train', 'test']:
            set_folder = os.path.join(class_folder, set_type)
            
            if not os.path.exists(set_folder):
                continue
            
            # Duyệt qua good/bad folders
            for label_type in ['good', 'bad']:
                label_folder = os.path.join(set_folder, label_type)
                
                if not os.path.exists(label_folder):
                    continue
                
                # Lấy danh sách các file image
                image_files = [f for f in os.listdir(label_folder) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                for image_file in image_files:
                    # Tạo đường dẫn tương đối
                    image_path = os.path.join(data_class, set_type, label_type, image_file)
                    
                    # Đường dẫn mask (thường trong ground_truth folder)
                    mask_name = image_file.replace('.JPG', '.png').replace('.jpg', '.png')
                    mask_path = os.path.join(data_class, 'ground_truth', label_type, mask_name)
                    
                    # Chuyển đổi label
                    csv_label = 'normal' if label_type == 'good' else 'anomaly'
                    
                    rows.append([data_class, set_type, csv_label, image_path, mask_path])
    
    # Ghi file CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['object', 'set', 'label', 'image_path', 'mask_path'])
        writer.writerows(rows)
    
    print(f"Created {output_csv} with {len(rows)} entries")

# Sử dụng
if __name__ == "__main__":
    data_folder = "./dataset/VisA_20220922"  # Điều chỉnh đường dẫn
    output_csv = "./split_csv/1cls.csv"
    
    create_visa_split_csv(data_folder, output_csv)