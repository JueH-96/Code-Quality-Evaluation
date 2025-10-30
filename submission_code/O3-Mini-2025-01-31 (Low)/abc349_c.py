def main():
    import sys
    input_lines = sys.stdin.read().strip().split()
    if not input_lines:
        return
    S = input_lines[0].strip()
    T = input_lines[1].strip()
    
    # Method 1: Subsequence of length 3, converting to uppercase
    def method1():
        t_index = 0
        for ch in S:
            if ch == T[t_index].lower():
                t_index += 1
                if t_index == 3:
                    return True
        return False

    # Method 2: Subsequence of length 2, converting to uppercase and appended with 'X'
    def method2():
        # T must end with 'X' for method2 to apply.
        if T[2] != 'X':
            return False
        t_index = 0
        for ch in S:
            if ch == T[t_index].lower():
                t_index += 1
                if t_index == 2:
                    return True
        return False

    if method1() or method2():
        sys.stdout.write("Yes")
    else:
        sys.stdout.write("No")

if __name__ == '__main__':
    main()