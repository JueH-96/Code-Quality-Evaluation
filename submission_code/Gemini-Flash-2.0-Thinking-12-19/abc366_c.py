def solve():
    q = int(input())
    bag_contents = {}
    for _ in range(q):
        query_parts = input().split()
        query_type = int(query_parts[0])
        if query_type == 1:
            x = int(query_parts[1])
            bag_contents[x] = bag_contents.get(x, 0) + 1
        elif query_type == 2:
            x = int(query_parts[1])
            count_x = bag_contents[x]
            count_x -= 1
            if count_x > 0:
                bag_contents[x] = count_x
            else:
                del bag_contents[x]
        elif query_type == 3:
            print(len(bag_contents))

if __name__ == '__main__':
    solve()