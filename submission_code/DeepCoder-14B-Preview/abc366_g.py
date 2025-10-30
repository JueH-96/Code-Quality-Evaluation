import sys

def main():
    q = int(sys.stdin.readline())
    count_map = dict()
    unique_set = set()
    for _ in range(q):
        parts = sys.stdin.readline().split()
        if parts[0] == '1':
            x = int(parts[1])
            if x in count_map:
                count_map[x] += 1
            else:
                count_map[x] = 1
                unique_set.add(x)
        elif parts[0] == '2':
            x = int(parts[1])
            current = count_map[x]
            current -= 1
            if current == 0:
                unique_set.discard(x)
                del count_map[x]
            else:
                count_map[x] = current
        elif parts[0] == '3':
            print(len(unique_set))
        else:
            pass  # This case should not occur as per problem constraints

if __name__ == '__main__':
    main()