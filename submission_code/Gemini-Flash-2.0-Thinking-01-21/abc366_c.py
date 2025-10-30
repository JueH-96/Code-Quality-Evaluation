def solve():
    q = int(input())
    bag_counts = {}
    for _ in range(q):
        query_line = input().split()
        query_type = query_line[0]
        if query_type == '1':
            x = int(query_line[1])
            bag_counts[x] = bag_counts.get(x, 0) + 1
        elif query_type == '2':
            x = int(query_line[1])
            bag_counts[x] -= 1
            if bag_counts[x] == 0:
                del bag_counts[x]
        elif query_type == '3':
            print(len(bag_counts))

if __name__ == '__main__':
    solve()