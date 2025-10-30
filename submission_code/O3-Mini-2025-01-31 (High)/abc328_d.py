def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    S = data[0]
    stack = []
    for ch in S:
        stack.append(ch)
        # If the last three characters form "ABC", remove them.
        if len(stack) >= 3 and stack[-3] == 'A' and stack[-2] == 'B' and stack[-1] == 'C':
            stack.pop()
            stack.pop()
            stack.pop()
    # Join the remaining characters and print the result.
    sys.stdout.write("".join(stack))
    
# Call the main function to execute the solution.
if __name__ == '__main__':
    main()