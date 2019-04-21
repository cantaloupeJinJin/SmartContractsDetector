pragma solidity ^0.4.25;

contract Fund {
	/// Mapping of ether shares of the contract.
	mapping(address => uint) shares;
	/// Withdraw your share.
	function withdraw() {
		if (msg.sender.send(shares[msg.sender]))
			shares[msg.sender] = 0;
	}
}