def find_kth_321_like_number(k):
    result = []
    for i in range(9, 0, -1):
        result.append(i)
        dfs(i, result, k)
        if len(result) >= k:
            return result[k-1]
    return -1

def dfs(num, result, k):
    if len(result) >= k:
        return
    for i in range(num-1, 0, -1):
        new_num = num * 10 + i
        result.append(new_num)
        dfs(i, result, k)

if __name__ == "__main__":
    k = int(input())
    print(find_kth_321_like_number(k))