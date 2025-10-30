def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s = input_data[0]
    stack = []
    
    # Process each character and check if the ending forms "ABC"
    for ch in s:
        stack.append(ch)
        if len(stack) >= 3 and stack[-3:] == ['A', 'B', 'C']:
            stack.pop()
            stack.pop()
            stack.pop()
    
    # The final string after removing the occurrences of "ABC"
    sys.stdout.write("".join(stack))

if __name__ == '__main__':
    main()