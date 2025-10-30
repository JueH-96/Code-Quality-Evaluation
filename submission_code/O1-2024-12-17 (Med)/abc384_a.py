def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    c1 = data[1]
    c2 = data[2]
    S = data[3]
    
    # Replace characters in S that are not c1 with c2
    result = ''.join(char if char == c1 else c2 for char in S)
    
    # Print the result
    print(result)

# Do not forget to call main.
main()