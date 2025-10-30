# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split('
')
    N = int(data[0])
    S = data[1]
    
    stack = []
    for c in S:
        if c != ')':
            stack.append(c)
        else:
            temp = []
            while stack:
                top = stack.pop()
                if top == '(':
                    if not any(ch in '()' for ch in temp):
                        # Simple pair, delete
                        break
                    else:
                        # Not simple, keep '(' and temp
                        stack.append('(')
                        stack.extend(temp[::-1])
                        stack.append(')')
                        break
                else:
                    temp.append(top)
            else:
                # No '(', keep ')'
                stack.append(')')
    print(''.join(stack))

if __name__ == "__main__":
    main()