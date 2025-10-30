class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        arr = [i for i in range(1, n + 1)]
        seen = set()
        i, j = 0, n - 1
        while i < j:
            s = arr[i] + arr[j]
            if s == k:
                arr[j] = k - arr[i] - 1
                j -= 1
                while (k - arr[j] in seen) or (arr[j] >= k - arr[i]):
                    arr[j] -= 1
                seen.add(arr[j])
            i += 1
            while (k - arr[i] in seen) or (k - arr[i] <= arr[i]):
                arr[i] += 1
                j -= 1
            
        return sum(arr)