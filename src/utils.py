import sys

def read_data(filename):
    """
    Reads dataset from 'data.txt' and parses feature coordinates and labels.
    Returns:
        list of ((float, float), float): A list where each element is a tuple 
        containing the input vector (x1, x2) and its associated class label.
    """
    data = []  
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                line_data = line.split(" ")
                if len(line_data) != 3:
                    print(f"\n[ERROR] Invalid file format at line {line_num}.")
                    print(f"Expected 3 values, found {len(line_data)}: '{line}'")
                    print("Format must be: x1 x2 C")
                    sys.exit(1)
                try:
                    x1 = float(line_data[0])
                    x2 = float(line_data[1])
                    category = float(line_data[2])
                    if category not in [1.0, -1.0]:
                        print(f"[ERROR] Line {line_num}: Category is {int(category)}. Expected 1 or -1.")
                        sys.exit(1)
                    data.append(((x1, x2), category))
                except ValueError:
                    print(f"\n[ERROR] Invalid data type at line {line_num}.")
                    print(f"Could not convert data to numbers: '{line}'")
                    sys.exit(1)
    except FileNotFoundError:
        print(f"\n[ERROR] File '{filename}' not found.")
        print("Please ensure the data file is in the same directory as the script.")
        sys.exit(1)
    return data