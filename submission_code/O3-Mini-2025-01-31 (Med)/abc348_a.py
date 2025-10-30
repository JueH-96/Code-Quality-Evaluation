def main():
    import sys
    input_data = sys.stdin.read().strip()
    if not input_data:
        return
    # Parse input
    N = int(input_data)
    
    result = ""
    for i in range(1, N + 1):
        if i % 3 == 0:
            result += "x"
        else:
            result += "o"
    
    print(result)

if __name__ == '__main__':
    main()