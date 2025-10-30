def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    maximum = max(A)
    filtered = [x for x in A if x != maximum]
    print(max(filtered))

# Do not remove the following function call
main()