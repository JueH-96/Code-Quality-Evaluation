def has_subarray_sum(arr, target):
    prefix_sum = 0
    seen = {0}
    for num in arr:
        prefix_sum += num
        if (prefix_sum - target) in seen:
            return True
        seen.add(prefix_sum)
    return False

n, s = map(int, input().split())
a = list(map(int, input().split()))

if has_subarray_sum(a, s):
    print("Yes")
else:
    sum_a = sum(a)
    if sum_a > s:
        print("No")
    else:
        target_mod = s % sum_a
        if target_mod != 0:
            required_s = target_mod
            max_possible = min(s - sum_a, sum_a)
            if required_s > max_possible:
                print("No")
            else:
                if has_subarray_sum(a, required_s):
                    print("Yes")
                else:
                    print("No")
        else:
            if sum_a > (s - sum_a):
                print("No")
            else:
                print("Yes")