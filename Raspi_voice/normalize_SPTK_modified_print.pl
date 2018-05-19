#!/usr/bin/perl

#
# Normalize features. Mean value 0 and valiable 1.0.
#

print STDERR "Normalizing feature vectors ... \n";

# read feature vectors
undef %frametbl;
$prevspk = "";
@buf = ();
$line = 0;
$numf = 0;
while(<>)
{
    chop;    
    $buf[$line++] = $_;
    $_ = <STDIN>; chop; $_ =~ s/^\s+//g;
    if(/^[-.0-9]/){ $numf++; } #count frame
    $buf[$line++] = $_;	
}

($mean, $std, $dim) = &mean_var();
&normalize();
print STDEE "Normalization finish.\n";
# end

# calculate mean and variance
sub mean_var{
    my $tot = 0;
    my $cnt = 0;
    my $dim = 0;	
    for(my $i=0; $i < $line; $i++)
    {
	$dim = 0;
	#print "guhe: $buf[$i]\n";
	foreach $val (split(/\s+/,$buf[$i]))
	{
	    if($val =~ /^[-.0-9]/)
	    {
		$tot += $val; $cnt++; $dim++;
	    }
	}
    }
    my $mean = $tot / $cnt; # mean
    $tot = 0;
    for(my $i=0; $i < $line; $i++)
    {
	foreach $val (split(/\s+/,$buf[$i]))
	{
	    if($val =~ /^[-.0-9]/)
	    {
		my $diff = $val - $mean;
		$tot = $tot + $diff * $diff;
	    }
	}
    }
    my $var = $tot / $cnt; # variance
    my $std = sqrt($var);  # std
    #print STDERR "m=$mean, sd=$std\n";
    return ($mean, $std, $dim);
}

# Function:
#   Normalization and output to file in binary format
# Format:
#   MFCC(13) Delta(13) Delta-Delta(13)
sub normalize{
    #print STDERR "speaker $prevspk\n";	
    #my $fnum = -1;
    for(my $i=0; $i < $line; $i++)
    {
	my $flg = 1;
	foreach $val (split(/\s+/,$buf[$i]))
	{
	    if($val =~ /^[-.0-9]/)
	    {
		$nval = ($val - $mean) / $std; # normalization
		print "$nval ";
	    }
	}
	if($i==$line-1){
	    print ""
	}
	else{
	    print "\n"
	}
    }
}
