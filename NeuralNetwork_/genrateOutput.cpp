#include<bits/stdc++.h>
using namespace std;
int main(){
	int n;
	n=1500;
	for(int i=0;i<n;i++){
		double a,b,c;
		cin>>a>>b>>c;
		b+=500;
		c+=500;
		double r2=(b-500)*(b-500)+(c-500)*(c-500);
		if(r2<=350*350)
			cout<<1<<endl;
		else
			cout<<0<<endl;
	}
}