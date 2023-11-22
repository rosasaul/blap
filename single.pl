#!/usr/bin/perl

use Math::Trig;

my $max_weight = 0.5;
my $max_iterations = 5000;
my $stop_threshold = 0.0001;

my $learning_rate = 0.1;
my $momentum = 0.05;

my %data;
readData('TEST',"data/iris_test.dat");
readData('TRAIN',"data/iris_training.dat");
readData('VALIDATE',"data/iris_validation.dat");

# initialize weight matrix
my $w1 = initMatrix(3,5);
#pMatrix($w1);
my $bias1 = 1;

my $epoch_count = 0;

my $train_count = 1 * @{ $data{'TRAIN'} };

while($epoch_count < $max_iterations){
  # Select random training row
  my $sample_index = int(rand($train_count));

  my @train_vector;
  foreach my $val (sort keys %{$data{'TRAIN'}[$sample_index]{'data'}}){
    push @train_vector, $data{'TRAIN'}[$sample_index]{'data'}{$val};
  }
  push @train_vector, $bias1;
#  print STDERR "first train_vector\n";
#  pVector(\@train_vector);

  #  net = train_item * weights
  my @out_vec = mulVecMat(\@train_vector, $w1);
#  print STDERR "out_vec\n";
#  pVector(\@out_vec);

  my @out_vec_derivative;
  for(my $i = 0; $i < @out_vec; $i++){
    $out_vec_derivative[$i] = sigmoid_derivative($out_vec[$i]);
    $out_vec[$i] = sigmoid($out_vec[$i]);
  }

  # Back propegation
  for(my $i = 0; $i < 3; $i++){
    $errorW1[$i] = $out_vec[$i] - $data{'TRAIN'}[$sample_index]{'label'}{$i};
  }
#  pVector(\@errorW1); 
  
  my @delta_w1;
  for(my $i = 0; $i < 3; $i++){
    $delta_w1[$i] = $errorW1[$i] * $out_vec_derivative[$i];
  }

  my $weight_delta_w1 = initMatrixZero(3,5);
#  pMatrix($weight_delta_w1);
  for(my $i = 0; $i < 3; $i++){
    for(my $j = 0; $j < 5; $j++){
      $weight_delta_w1->[$i][$j] = -$learning_rate * $train_vector[$j] * $delta_w1[$i];
    }
  }
#  pMatrix($weight_delta_w1);

  # Add Weight Delta to W1 layer
#  pMatrix($w1);
  for(my $i = 0; $i < 3; $i++){
    for(my $j = 0; $j < 6; $j++){
      $w1->[$i][$j] += $weight_delta_w1->[$i][$j];
    }
  }
#  pMatrix($w1);

  $epoch_count++;

  my $accuracy = check_Error($w1,$bias1);
  print STDERR "epoch $epoch_count accuracy ".sprintf("%.2f",$accuracy * 100)."%\n";
  if($accuracy eq 1){ last; }
  #if($prev_error - $error < $stop_threshold){ last; }
  #$prev_error = $error;
}

exit;

####### SUBS #######
sub check_Error{
  my ($w1,$bias1) = @_;
  my $total = 0; my $correct = 0;
  foreach my $item (@{ $data{'TEST'} }){
    my @vector;
    foreach my $val (sort keys %{$item->{'data'}}){
      push @vector, $item->{'data'}{$val};
    }
    push @vector, $bias1;
    #pVector(\@vector); die;

    my @out_vec = mulVecMat(\@vector, $w1);
    for(my $i = 0; $i < @out_vec; $i++){
      $out_vec[$i] = sigmoid($out_vec[$i]);
    }
    #pVector(\@out_vec); die;

#    if($item->{'label'}{0}){ print STDERR "label 0"; }
#    elsif($item->{'label'}{1}){ print STDERR "label 1"; }
#    elsif($item->{'label'}{2}){ print STDERR "label 2"; }

    if($item->{'label'}{0} eq 1 and $out_vec[0] > $out_vec[1] and $out_vec[0] > $out_vec[2]){ $correct++; }
    elsif($item->{'label'}{1} eq 1 and $out_vec[1] > $out_vec[0] and $out_vec[1] > $out_vec[2]){ $correct++; }
    elsif($item->{'label'}{2} eq 1 and $out_vec[2] > $out_vec[1] and $out_vec[2] > $out_vec[0]){ $correct++; }
    $total++;
  }
  return $correct / $total;
}
sub pMatrix{
  my $matrix = $_[0];
  my $output = "::Matrix::\n";
  for(my $i = 0; $i < @{$matrix}; $i++){
    $output .= "|";
    for(my $j = 0; $j < @{$matrix->[$i]}; $j++){
      $output .= " ".$matrix->[$i][$j];
    }
    $output .= " |\n";
  }
  $output .= "::Matrix::\n\n";
  print STDERR $output;
}
sub pVector{
  my $vector = $_[0];
  my $output = "::Vector::\n[";
  for(my $i = 0; $i < @{$vector}; $i++){
    $output .= " ".$vector->[$i];
  }
  $output .= "]\n::Vector::\n\n";
  print STDERR $output;
}
sub sigmoid{
  my $x = $_[0];
  return ((tanh($x) + 1) / 2);
}
sub sigmoid_derivative{
  my $x = $_[0];
  return ((1 - (tanh($x) * tanh($x)))/2);
}
sub mulVecMat{
  my $vector = $_[0];
  my $matrix = $_[1];
  my @output;
  for(my $i = 0; $i < @{$matrix}; $i++){

    for(my $j = 0; $j < @{$vector}; $j++){
      $output[$i] += $vector->[$j] * $matrix->[$i][$j];
    }

  }
  return @output;
}
sub initMatrix{
  my $rows = $_[0];
  my $cols = $_[1];
  my @matrix;
  for(my $i = 0; $i < $rows; $i++){
    for(my $j = 0; $j < $cols; $j++){
      $matrix[$i][$j] = $max_weight * rand();
    }
  }
  return \@matrix;
}
sub initMatrixZero{
  my $rows = $_[0];
  my $cols = $_[1];
  my @matrix;
  for(my $i = 0; $i < $rows; $i++){
    for(my $j = 0; $j < $cols; $j++){
      $matrix[$i][$j] = 0;
    }
  }
  return \@matrix;
}

sub readData {
  my $mode = $_[0];
  my $file = $_[1];
  my @set;
  
  open IN, "<".$file or die "Failed to open $file : $!\n";
  while(my $line = <IN>){ chomp($line);
    #print STDERR "line : $line\n";
    my @items = split(/\s+/,$line);
    my %item;
    $item{'data'}{0} = $items[0];
    $item{'data'}{1} = $items[1];
    $item{'data'}{2} = $items[2];
    $item{'data'}{3} = $items[3];

    $item{'label'}{0} = $items[4];
    $item{'label'}{1} = $items[5];
    $item{'label'}{2} = $items[6];
    push @set, \%item;
  }
  close IN;
  $data{$mode} = \@set;
}
