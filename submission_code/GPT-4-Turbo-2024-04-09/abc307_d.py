def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    result = []
    balance = 0
    
    for char in S:
        if char == '(':
            balance += 1
            result.append(char)
        elif char == ')':
            if balance > 0:
                balance -= 1
                result.append(char)
        else:
            result.append(char)
    
    # Remove excess opening brackets from the end of the result
    final_result = []
    open_count = 0
    for char in reversed(result):
        if char == '(' and open_count >= balance:
            balance -= 1
        else:
            final_result.append(char)
            if char == '(':
                open_count += 1
    
    print(''.join(reversed(final_result)))

if __name__ == "__main__":
    main()