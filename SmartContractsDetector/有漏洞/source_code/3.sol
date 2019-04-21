pragma solidity ^0.4.25;

contract StandardToken is ERC20, BasicToken {
    
    function transferFrom(address _from, address _to, uint _value) returns (bool) {
        uint _allowance = allowed[_from][msg.sender];
        require(_allowance >= _value);
        balances[_to] = balances[_to].add(_value);
        balances[_from] = balances[_from].sub(_value);
        allowed[_from][msg.sender] = _allowance.sub(_value);
        emit Transfer(_from, _to, _value);
    }
}