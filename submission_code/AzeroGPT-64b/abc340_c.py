# coding: utf-8

# In[1]:


def solve(n):
    ans = 0
    while n > 1:
        dit = 1
        while n >= dit * 2:
            dit *= 2
        ans += n
        n = (n - dit) + dit // 2 + dit // 2
    return ans

n = int(input())
print(solve(n))