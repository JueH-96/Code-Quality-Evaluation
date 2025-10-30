# YOUR CODE HERE

def solve():
    count = 0
    for i in range(1, 13):
        if len(input()) == i:
            count += 1
    print(count)

if __name__ == "__main__":
    solve()