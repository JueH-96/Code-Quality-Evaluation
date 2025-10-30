def solve():
    n = int(input())
    s = input()
    found_pair = False
    for i in range(n - 1):
        pair_set = {s[i], s[i+1]}
        if pair_set == {'a', 'b'}:
            found_pair = True
            break
    if found_pair:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()