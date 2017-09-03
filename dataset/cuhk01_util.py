import os

def collect_data(root):
    file_list_a=os.listdir(root)
    name_dict={}
    for name in file_list_a:
        if name[-3:]=='png':
            id = name[:4]
            if not name_dict.has_key(id):
                name_dict[id]=[]
                name_dict[id].append([])
                name_dict[id].append([])
            if int(name[-7:-4])<3:
                name_dict[id][0].append(root+name)  
            else:
                name_dict[id][1].append(root+name)  
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
            new_lines.append(filename)
    lines=new_lines
    return lines