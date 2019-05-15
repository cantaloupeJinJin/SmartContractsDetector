contract Contract2 {

    Contract1 public original;
  
  	mapping (uint16 => mapping (address => uint8)) public something;

    // コンストラクタ
    function Contract2(address c) public {
        original = Contract1(c);
    }


	function test(uint8 x, address a)public{
		if(original.something(uint8(x),a))
			something[x][a] = 1;
		else
			something[x][a] = 2;
	}
}