#include<bits/stdc++.h>
using namespace std;
int main(){
	int n;
	n=1500;
	srand(time(NULL));
	for(int i=0;i<n;i++){
		int x=rand()%1001;
		int y=rand()%1001;
		cout<<1<<" "<<x-500<<" "<<y-500<<endl;
		
	}
	
}