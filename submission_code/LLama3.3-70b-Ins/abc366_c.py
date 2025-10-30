import sys

def solve():
    Q = int(input())
    bag = set()
    for _ in range(Q):
        query = input().split()
        if query[0] == '1':
            bag.add(int(query[1]))
        elif query[0] == '2':
            bag.remove(int(query[1]))
        else:
            print(len(bag))

if __name__ == '__main__':
    solve()