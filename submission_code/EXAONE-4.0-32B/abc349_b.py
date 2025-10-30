def main():
    S = input().strip()
    n = len(S)
    freq_map = {}
    for char in S:
        freq_map[char] = freq_map.get(char, 0) + 1
        
    arr = [0] * (n + 1)
    
    for count in freq_map.values():
        if count <= n:
            arr[count] += 1
            
    for i in range(1, n + 1):
        if arr[i] != 0 and arr[i] != 2:
            print("No")
            return
            
    print("Yes")

if __name__ == "__main__":
    main()