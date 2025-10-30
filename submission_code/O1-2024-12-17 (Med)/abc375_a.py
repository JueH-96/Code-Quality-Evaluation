def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    
    count = 0
    for i in range(N - 2):
        if S[i] == '#' and S[i+1] == '.' and S[i+2] == '#':
            count += 1
    
    print(count)

# Call main function
main()