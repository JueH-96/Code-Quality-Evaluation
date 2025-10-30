T=int(input())
for i in range(T):
    K,s_x,s_y,t_x,t_y=list(map(int,input().split()))
    S_x_s=S_x_t=S_y_s=S_y_t=n1=n2=0
    if s_x%K<0:
        S_x_s=abs(s_x--s_x%K)
        S_x_t=abs(--s_x-s_x%K)
    if s_x%K==0:
        S_x_s=abs(s_x-s_x+K)
        S_x_t=abs(s_x-s_x+K)
    if s_x%K>0 and s_x%K<K:
        S_x_s=abs(s_x--s_x%K)
        S_x_t=abs(s_x-s_x%K)
    if t_x%K<0:
        S_y_s=abs(t_x--t_x%K)
        S_y_t=abs(--t_x-t_x%K)
    if t_x%K==0:
        S_y_s=abs(t_x-t_x+K)
        S_y_t=abs(t_x-t_x+K)
    if t_x%K>0 and t_x%K<K:
        S_y_s=abs(t_x--t_x%K)
        S_y_t=abs(t_x-t_x%K)
    top1=0
    if (s_x//K)%2==0:
        if (s_y//K)%2==0:
            top1=(s_y-s_y%K-detail)/(K)+(s_x-s_x%K+S_x_s)/(K)
            top2=(s_y-s_y%K+detail)/(K)+(s_x-s_x%K+S_x_t)/(K)
        if (s_y//K)%2==1:
            top1=(s_y-s_y%K)/(K)+(s_x-s_x%K+S_x_s)/(K)
            top2=(s_y-s_y%K+1)/(K)+(s_x-s_x%K+S_x_t)/(K)
    elif (s_x//K)%2==1:
        if (s_y//K)%2==0:
            top1=(s_y-s_y%K)/(K-1)+(s_x-s_x%K+S_x_s)/K
            top2=(s_y-s_y%K+detail)/(K-1)+(s_x-s_x%K+S_x_t)/K
        if (s_y//K)%2==1:
            top1=(s_y-s_y%K)/(K-1)+(s_x-s_x%K+S_x_s)/K
            top2=(s_y-s_y%K+1)/(K-1)+(s_x-s_x%K+S_x_t)/K
    
    print(round((s_x-t_x)**2+(s_y-t_y)**2,0))