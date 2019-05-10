#execute: vmd -dispdev text -e water_count.tcl -args FOLDER_NAME > pbc_log.txt (eg: 6b73B)

set code $argv

mol new ../../$code/results/step5_assembly.xplor_ext.psf type psf
mol off top
set first_frame 0
set last_frame -1

set filelist [glob ../../$code/results/namd/step7.*_production.dcd-pbc.dcd]
#set filelist [glob ../../step7.*_production.dcd-pbc.dcd]
set nf [llength $filelist]

set x 10
set y 18
set z 10
set _x 17
set _y 18
set _z 17
set distance 5

set selection [atomselect top "(water within $distance of protein) and (x<$x and y<$y and z<$z and -x<$_x and -y<$_y and -z<$_z)"]

for { set i 1 } { $i <= $nf } { incr i } {
	set crnt_file [glob ../../$code/results/namd/step7.${i}_production.dcd-pbc.dcd]

	animate read dcd $crnt_file beg 0 end -1 waitfor all

	set num_steps [molinfo top get numframes]
        
	set out_file $crnt_file-warter_count.txt
	set fid [open $out_file w]

	for {set frame 0} {$frame < $num_steps} {incr frame} {
		#Update the frame
		$selection frame $frame
		$selection update
		
		#Count atoms in selection
		set found_atoms [$selection num]
		
		#compose frame number to write to the file
		set frame_write [format "%04d" [expr {$frame + $first_frame}]]	
			
		#Write results into file
		puts $fid "$frame_write\t$found_atoms"
		
#		$selection delete
		puts "global frame [expr {$frame + $first_frame}]     file $crnt_file       frame $frame of [expr {($num_steps - 1)}] finished"
	}
	#Close written files
	close $fid

	set first_frame [expr {$first_frame + $num_steps}]
	animate delete all
}

puts "Done"
quit


