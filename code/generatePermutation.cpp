#include<bits/stdc++.h>
using namespace std;
#define ll long long


/*
The code is used for generate permutation of synthetic workloads for ContTune.
*/

int main(){
    srand(time(NULL));
    set<int> st;
    printf("order = \n");
    while(st.size()!=10)
    {
        int tmp = (rand() % 10) + 1;
        if(st.count(tmp)) continue;
        else
        {
            printf("%2d * unitWorkload\n",tmp);
            st.insert(tmp);
        }
    }
    return 0;
}
