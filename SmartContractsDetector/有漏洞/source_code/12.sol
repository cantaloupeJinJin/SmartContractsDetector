pragma solidity ^0.4.25;

contract StandardToken is ERC20, BasicToken {

	function transferFrom(address _from, address _to, uint256 _value) returns (bool) {
		var _allowance = allowed[_from][msg.sender];
		balances[_to] = balances[_to].add(_value);
		balances[_from] = balances[_from].sub(_value);
		allowed[_from][msg.sender] = _allowance.sub(_value);
		Transfer(_from, _to, _value);
		return true;
	}
	
}