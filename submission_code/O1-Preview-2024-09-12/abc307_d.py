# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().rstrip()

    stack = []
    for c in S:
        if c.isalpha():
            stack.append(('letter', c))
        elif c == '(':
            stack.append(('(', ))
        elif c == ')':
            temp = []
            can_remove = True
            while stack:
                top = stack.pop()
                if top[0] == 'letter':
                    temp.append(top)
                elif top[0] == '(':
                    # Found matching '(', check if all popped are letters
                    # They are, since we only push letters and '(' onto stack
                    # So we can remove this substring
                    break
                else:
                    # Found another ')', invalid, cannot remove
                    can_remove = False
                    temp.append(top)
                    break
            else:
                # Stack empty, no matching '(', cannot remove
                can_remove = False

            if not can_remove:
                # Put back all popped items
                for item in reversed(temp):
                    stack.append(item)
                # Push the ')'
                stack.append((')', ))
            # If can_remove, we do not put back the popped items
    # Build the result
    result = ''.join(item[1] if item[0]=='letter' else item[0] for item in stack)
    print(result)


if __name__ == '__main__':
    threading.Thread(target=main).start()