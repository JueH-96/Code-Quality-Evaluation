import functools

def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    persons = []
    for i in range(1, n + 1):
        line = data[i].split()
        a = int(line[0])
        b = int(line[1])
        persons.append((a, b, i))
    
    def cmp_func(x, y):
        a1, b1, idx1 = x
        a2, b2, idx2 = y
        left = a1 * (a2 + b2)
        right = a2 * (a1 + b1)
        if left > right:
            return -1
        elif left < right:
            return 1
        else:
            if idx1 < idx2:
                return -1
            elif idx1 > idx2:
                return 1
            else:
                return 0
                
    sorted_persons = sorted(persons, key=functools.cmp_to_key(cmp_func))
    result = [str(person[2]) for person in sorted_persons]
    print(" ".join(result))

if __name__ == '__main__':
    main()