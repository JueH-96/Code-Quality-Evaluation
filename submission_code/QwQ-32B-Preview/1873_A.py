def can_get_abc(s):
    if s == "abc":
        return "YES"
    else:
        swap1 = swap(s, 0, 1)
        swap2 = swap(s, 0, 2)
        swap3 = swap(s, 1, 2)
        if swap1 == "abc" or swap2 == "abc" or swap3 == "abc":
            return "YES"
        else:
            return "NO"

def swap(s, i, j):
    lst = list(s)
    lst[i], lst[j] = lst[j], lst[i]
    return ''.join(lst)

def main():
    import sys
    input_lines = sys.stdin.read().splitlines()
    t = int(input_lines[0])
    for i in range(1, t+1):
        s = input_lines[i]
        print(can_get_abc(s))

if __name__ == "__main__":
    main()