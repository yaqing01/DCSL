import os

def file_to_id(file_obj):
    try:
        all_the_text = file_obj.read( )
    finally:
        file_obj.close( )

    lines = all_the_text.split('\n')
    new_lines = []
    for filename in lines:
        if filename!='' :
            campair_no = int(filename.split(',')[0])
            person_id = int(filename.split(',')[1])
            #print int(campair_no)
            if campair_no<=3:                
                new_lines.append(filename)
    return new_lines

def collect_data(root):
    
    
    filename_train = root + 'exp_set/set01_train_noval.txt'
    filename_test = root + 'exp_set/set01_test_noval.txt'
    
    file_object = open(filename_train)
    key_list = file_to_id(file_object)
    
    file_object = open(filename_test)
    key_list.extend(file_to_id(file_object))
    
    name_dict={}
    
    for filename in key_list:
        if not name_dict.has_key(filename):
            name_dict[filename] = []
            name_dict[filename].append([])
            name_dict[filename].append([])
            
        if filename!='' :
            campair_no = int(filename.split(',')[0])
            person_id = int(filename.split(',')[1])
            #print int(campair_no)
            for img_no in range(5):
                this_filename = root  +  'data/campair_%d/'%campair_no + '%02d_%04d_%02d.jpg'%(campair_no,person_id,img_no+1)
                if os.path.isfile(this_filename):
                    name_dict[filename][0].append(this_filename)
            for img_no in range(5):
                this_filename = root  +  'data/campair_%d/'%campair_no + '%02d_%04d_%02d.jpg'%(campair_no,person_id,img_no+6)
                if os.path.isfile(this_filename):
                    name_dict[filename][1].append(this_filename)                

    return name_dict

def partition_file_to_list(path):
    file_object = open(path)
    try:
        all_the_text = file_object.read( )
    finally:
        file_object.close( )
        
    lines = all_the_text.split('\n')
    new_lines = []
    for filename in lines:
        if filename!='' :
            campair_no = int(filename.split(',')[0])
            person_id = int(filename.split(',')[1])
            #print int(campair_no)
            if campair_no<=3: 
                new_lines.append(filename)
    lines=new_lines
    return lines