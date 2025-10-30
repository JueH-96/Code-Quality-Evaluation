def main():
    s = input().strip()
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pos_dict = {}
    for idx, char in enumerate(s):
        pos_dict[char] = idx + 1
    
    coords = [pos_dict[char] for char in alphabet]
    
    total_distance = 0
    for i in range(1, 26):
        total_distance += abs(coords[i] - coords[i-1])
        
    print(total_distance)

if __name__ == "__main__":
    main()