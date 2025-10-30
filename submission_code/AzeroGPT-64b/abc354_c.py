import sys
from bisect import bisect_right

def main():
    input = sys.stdin.buffer.readline
    N = int(input())
    AC = [tuple(map(int, input().split())) for _ in range(N)]
    
    # Aが昇順になるようにカードをソートします
    AC.sort(key=lambda x: x[0])
    As = [a for a, c in AC]

    # 対応するAのインデックスを管理するリスト
    idxs = []
    vals = [10**9 + 1] # 最初のCより大きな値を設定
    
    # 分解整合探索木（ここではソート済みリストを使って等価なことをします）
    for a, c in AC:
        # binary search. Find the largest C in vals that is less than C of the current card
        i = bisect_right(vals, c)
        
        # If the current card's C is greater than any C in vals, add it to the end
        if i == len(vals):
            idxs.append((len(vals)-1, a))
            vals.append(c)
        else:  # otherwise, replace the C with the current one
            idxs.append((i-1, a))
            vals[i] = c
    
    # Get the cards that are not eliminated
    ans = []
    val_idx = len(vals) - 2  # Start from the second last card in vals
    for i, a in idxs[::-1]:  # Traverse AC in reverse order
        if i == val_idx:
            ans.append(a)
            val_idx -= 1
    
    print(len(ans))
    print(*sorted(ans))

if __name__ == '__main__':
    main()