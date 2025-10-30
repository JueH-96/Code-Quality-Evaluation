#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n;
    cin >> n;
    vector<ll> A(n);
    for (int i = 0; i < n; i++){
        cin >> A[i];
    }
    
    // For each starting slime we “simulate” an absorption process.
    // We consider two orders:
    //   Order 0: always try to absorb the left neighbor first, then the right.
    //   Order 1: always try to absorb the right neighbor first, then the left.
    // (When both neighbors are available the order does not matter,
    // but if only one direction is available (or one side’s barrier
    // blocks further absorption) the order can affect the final size.)
    vector<ll> ans(n, 0);
    for (int k = 0; k < n; k++){
        ll base = A[k];
        ll best = base; // at worst, Takahashi does nothing.
        for (int order = 0; order < 2; order++){
            ll S = base;
            int L = k, R = k;
            bool changed = true;
            // repeatedly absorb while a neighbor is absorbable.
            while(changed) {
                changed = false;
                if(order == 0) { // try left then right
                    if(L > 0 && A[L-1] < S){
                        S += A[L-1];
                        L--;
                        changed = true;
                    }
                    if(R < n-1 && A[R+1] < S){
                        S += A[R+1];
                        R++;
                        changed = true;
                    }
                } else { // try right then left
                    if(R < n-1 && A[R+1] < S){
                        S += A[R+1];
                        R++;
                        changed = true;
                    }
                    if(L > 0 && A[L-1] < S){
                        S += A[L-1];
                        L--;
                        changed = true;
                    }
                }
            }
            best = max(best, S);
        }
        ans[k] = best;
    }
    
    // print answers B1 B2 ... BN separated by spaces
    for (int i = 0; i < n; i++){
        cout << ans[i] << (i < n-1 ? " " : "
");
    }
    return 0;
}