def main():
    import sys
    data = sys.stdin.read().strip()
    N = int(data)
    
    result = []
    for i in range(1, N + 1):
        if i % 3 == 0:
            result.append('x')
        else:
            result.append('o')
    
    print(''.join(result))

# Do not remove the below function call
main()