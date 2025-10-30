def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    A = list(map(int, input_data))
    
    # The sorted target for permutation of (1,2,3,4,5)
    target = [1, 2, 3, 4, 5]
    
    # Check each possible adjacent swap:
    possible = False
    for i in range(4):  # indices 0 to 3 for adjacent pairs in 5-element list
        temp = A.copy()
        # swap adjacent elements at positions i and i+1
        temp[i], temp[i+1] = temp[i+1], temp[i]
        if temp == target:
            possible = True
            break
    
    print("Yes" if possible else "No")
    
if __name__ == '__main__':
    main()