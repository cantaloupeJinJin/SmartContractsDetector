pragma solidity ^0.4.25;

contract GreaterOrEqualToZero {
	function infiniteLoop() {
		for (var i = 100; i >= 0; i--){
			...
		}
	}
}
