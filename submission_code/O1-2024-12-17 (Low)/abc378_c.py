def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    last_occ = {}
    B = []
    
    for i in range(N):
        val = A[i]
        if val in last_occ:
            B.append(last_occ[val])
        else:
            B.append(-1)
        last_occ[val] = i+1  # Store the 1-based index
    
    print(" ".join(map(str, B)))

# Call the main function
main()