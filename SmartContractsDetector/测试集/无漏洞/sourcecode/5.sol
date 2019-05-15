/**
 * Source Code first verified at https://etherscan.io on Friday, May 3, 2019
 (UTC) */

pragma solidity ^0.4.18;


contract Contract1 {

	mapping (uint8 => mapping (address => bool)) public something;

	function settrue(uint8 x, address a)public{
		something[x][a] = true;
	}
	function setfalse(uint8 x, address a)public{
		something[x][a] = false;
	}
}



